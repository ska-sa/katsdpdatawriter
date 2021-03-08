#!groovy

@Library('katsdpjenkins@master') _

katsdp.killOldJobs()
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master',
                        'ska-sa/katdal/master',
                        'ska-sa/katsdpservices/master',
                        'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(push_external: true)
katsdp.mail('sdpdev+katsdpdatawriter@ska.ac.za')
